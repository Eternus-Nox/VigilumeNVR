"""Read the RECORDING back: full-resolution frames of a moment that has passed.

Used two ways. As LIVE BURSTS (native/burst.py): starting a second after a
person or vehicle is first seen, the seconds from just BEFORE it was detected
up to now are decoded at 10 fps and read frame by frame, while it is still in
view. And, for plates, once more after the vehicle leaves, over whatever the
bursts did not cover. The original design note, written for the second use:

WHY
===
The live plate path gets one or two looks at a passing car: a full-resolution
snapshot about once a second (every half second in a plate zone), each arriving
a few hundred milliseconds after the moment it was asked for, and only once the
car has been tracked. A car that crosses the view in two or three seconds is
often read from a single blurred or turned-away look — or not at all. Measured
offline the reader is right ~90% of the time on a clean look; live, a plate was
being read about one pass in three. The difference is the number of looks.

The camera's full-resolution stream is already on disk: the recorder copies it
continuously into 10 s segments at the camera's full frame rate. So when a
car's track ends without a settled plate, this decodes exactly the seconds the
car was in view, cropped to the path it drove, at ~6 frames a second — dozens
of looks, none of them late — and the plate pass reads every one. A plate
arrives on the event later than a live read would (the pass finishes a track
about a minute after it is lost), but it arrives.

NEVER RAISES, NEVER BLOCKS DETECTION
-----------------------------------
Everything here runs in a background task and in threads. No recording (the
camera does not record, or the footage has already been pruned), no ffmpeg, or
a decode that fails all mean "no frames", and the pass concludes on what it
already had. One replay runs at a time across all cameras, so a busy road
cannot turn this into a CPU storm.

WHEN A RECORDED FRAME WAS TAKEN
-------------------------------
A frame's wall time is its segment's start plus its offset in the segment.
Segment files are NAMED with whole seconds (``%M.%S.ts``), so a start read from
the name is early by the dropped fraction — up to a second, measured 0.37-0.87
s on a real-time stream-copy recording. A fast car moves its own length in
that time, and the frame would be matched to where it was a second earlier.
But the recorder cuts a segment the instant the next keyframe arrives: the
PREVIOUS segment's last write (its mtime) is the new one's start, to the
millisecond. `segment_starts` uses that when it is consistent with the name.

A segment still being written decodes cleanly except its very last frame,
which can be torn (measured: every earlier frame bit-exact, the last one not).
A live read drops it.

DECODED ON THE GPU WHEN THERE IS ONE
------------------------------------
Every frame of the window has to be decoded to pick six a second from it —
up to 15 s of the main stream, which on a 4K H.265 camera is hundreds of
frames and the most CPU this pass ever spends. With an NVIDIA card in the
container the decode runs on its NVDEC block (`-hwaccel cuda`; frames come
back to system memory for the crop, so nothing else changes). If that cannot
start, the same decode is retried on the CPU.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np

from .recorder import SEGMENT_SECONDS, select_segments
from .transcode import find_nvidia_device

log = logging.getLogger(__name__)

#: Frames per second decoded from the recording. Six looks a second is several
#: times what the live path gets, and past this neighbouring frames stop being
#: independent looks (see the time-diversity note in bestshot.py).
REPLAY_FPS = 6.0

#: Longest stretch of footage read for one vehicle. A car that parks and stays
#: tracked for an hour does not get an hour decoded: the plate is readable as
#: it arrives, and the first seconds are what matter.
MAX_WINDOW_S = 15.0

#: Padding before the first and after the last sighting, for the car entering
#: and leaving view before and after the tracker had it.
WINDOW_PAD_S = 0.7

#: Margin around the car's path when cropping the decoded frames, as a fraction
#: of the path box. The decode is cropped so a 4K frame is not held in memory
#: for a region a tenth of its size.
REGION_MARGIN = 0.15

#: A previous segment's mtime is taken as the next one's start only within this
#: of the named (truncated) start.
_MTIME_TRUST_S = 1.5

#: Consecutive failed GPU decodes after which replays decode on the CPU only.
HW_GIVE_UP = 3

#: Longest a single replay decode may take before it is abandoned.
DECODE_TIMEOUT_S = 60.0

_DIMS = re.compile(r"Stream #\d+:\d+.*?Video: .*?(\d{2,5})x(\d{2,5})")


@dataclass
class ReplayFrame:
    time: float
    crop: np.ndarray
    #: Origin of `crop` in the full frame, and the full frame's size.
    ox: int
    oy: int
    width: int
    height: int


class RecordingReplay:
    """Decodes a time window of a camera's recording. One per process,
    shared by the face and plate passes."""

    def __init__(
        self,
        camera_dir: Callable[[str], Path],
        ffmpeg: Optional[str] = None,
        *,
        fps: float = REPLAY_FPS,
        hwaccel: Optional[bool] = None,
    ) -> None:
        """`hwaccel`: decode on NVDEC first. None = when an NVIDIA GPU is in
        the container."""
        self._camera_dir = camera_dir
        self._ffmpeg = ffmpeg if ffmpeg is not None else shutil.which("ffmpeg")
        self._fps = fps
        self._hwaccel = find_nvidia_device() if hwaccel is None else bool(hwaccel)
        #: Decodes that produced frames, by how ("cuda" | "cpu"), since boot.
        self.decodes: dict[str, int] = {"cuda": 0, "cpu": 0}
        # Consecutive replays whose GPU decode produced nothing. A GPU that
        # cannot decode this footage is not asked again after HW_GIVE_UP.
        self._hw_misses = 0
        # One at a time, across every camera.
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return bool(self._ffmpeg)

    async def frames(
        self,
        camera: str,
        start: float,
        end: float,
        region: Sequence[float],
        *,
        fps: Optional[float] = None,
        pad: bool = True,
        live: bool = False,
    ) -> list[ReplayFrame]:
        """Recorded frames for [start, end] (wall-clock epoch), cropped to
        `region` (x1, y1, x2, y2 as FRACTIONS of the frame). [] on anything
        that prevents it. Never raises.

        `pad` widens the window by WINDOW_PAD_S each side (an after-the-fact
        replay of a whole track); a burst asks for exactly its window. `live`
        means the window may reach into the segment being written: its last
        decoded frame can be torn, so it is dropped. `fps` overrides the rate.
        """
        if not self._ffmpeg:
            return []
        if pad:
            start = start - WINDOW_PAD_S
            end = end + WINDOW_PAD_S
        end = min(end, start + MAX_WINDOW_S)
        if end <= start:
            return []
        rate = float(fps) if fps else self._fps
        async with self._lock:
            try:
                out = await asyncio.wait_for(
                    asyncio.to_thread(self._frames_blocking, camera, start, end, region, rate),
                    DECODE_TIMEOUT_S,
                )
                return out[:-1] if live and out else out
            except asyncio.TimeoutError:
                log.warning("plate replay on %s took over %.0f s — abandoned",
                            camera, DECODE_TIMEOUT_S)
                return []
            except Exception:  # noqa: BLE001
                log.exception("plate replay on %s failed", camera)
                return []

    # -- blocking work (threads) ------------------------------------------

    def _frames_blocking(
        self, camera: str, start: float, end: float, region: Sequence[float],
        fps: Optional[float] = None,
    ) -> list[ReplayFrame]:
        import subprocess

        fps = fps or self._fps
        # One segment further back than the window needs, so the first
        # useful segment's predecessor is there to time it (segment_starts).
        segments = segment_starts(
            select_segments(self._camera_dir(camera), start - SEGMENT_SECONDS, end)
        )
        first = 0
        for i, (seg_start, _) in enumerate(segments):
            if seg_start <= start:
                first = i
        segments = segments[first:]
        if not segments:
            return []
        dims = self._probe_dims(segments[0][1])
        if dims is None:
            return []
        width, height = dims
        x1, y1, x2, y2 = (float(v) for v in region[:4])
        mx, my = (x2 - x1) * REGION_MARGIN, (y2 - y1) * REGION_MARGIN
        cx1 = max(0, int((x1 - mx) * width)) & ~1
        cy1 = max(0, int((y1 - my) * height)) & ~1
        cx2 = min(width, int(round((x2 + mx) * width)))
        cy2 = min(height, int(round((y2 + my) * height)))
        cw, ch = (cx2 - cx1) & ~1, (cy2 - cy1) & ~1
        if cw < 16 or ch < 16:
            return []

        out: list[ReplayFrame] = []
        with tempfile.TemporaryDirectory(prefix="vigilume-replay-") as tmp:
            # Each segment's share of the window, decoded from THAT FILE with
            # the seek before its input — see decode_portions for why not
            # one decode over the joined segments.
            for k, (seg_start, seg_path, p_start, p_end) in enumerate(
                decode_portions(segments, start, end)
            ):
                pattern = Path(tmp) / f"p{k}_%04d.jpg"
                files: list[Path] = []
                for how, args in replay_attempts(
                    self._ffmpeg, seg_path, pattern, p_start - seg_start, p_end - p_start,
                    fps, (cw, ch, cx1, cy1), hwaccel=self._hwaccel,
                ):
                    for stale in Path(tmp).glob(f"p{k}_*.jpg"):
                        stale.unlink()
                    proc = subprocess.run(args, capture_output=True, timeout=DECODE_TIMEOUT_S)
                    files = sorted(Path(tmp).glob(f"p{k}_*.jpg"))
                    if files:
                        self.decodes[how] += 1
                        if how == "cuda":
                            self._hw_misses = 0
                        elif self._hwaccel:
                            self._note_hw_miss(camera, proc)
                        break
                    log.debug("replay decode attempt failed on %s (rc %s): %s", camera,
                              proc.returncode, proc.stderr.decode(errors="replace")[-300:])
                for i, path in enumerate(files):
                    img = cv2.imread(str(path))
                    if img is None:
                        continue
                    # Frame i of this portion is i/fps after its start.
                    out.append(ReplayFrame(time=p_start + i / fps, crop=img,
                                           ox=cx1, oy=cy1, width=width, height=height))
        return out

    def _note_hw_miss(self, camera: str, proc: Any) -> None:
        self._hw_misses += 1
        if self._hw_misses >= HW_GIVE_UP:
            self._hwaccel = False
            log.warning(
                "plate replay: GPU decoding failed %d times in a row (last on %s) — "
                "decoding replays on the CPU from now on", self._hw_misses, camera,
            )

    def status(self) -> dict[str, Any]:
        return {"available": self.available, "gpu_decode": self._hwaccel,
                "decodes": dict(self.decodes)}

    def _probe_dims(self, segment: Path) -> Optional[tuple[int, int]]:
        """(width, height) of a recorded segment, from ffmpeg's stream line."""
        import subprocess

        try:
            # Decode ONE frame to a null sink rather than giving ffmpeg an
            # input and no output: that prints the same stream line, exits 0,
            # and does not depend on how a given build handles "no output"
            # (one static build segfaults on it).
            proc = subprocess.run(
                [self._ffmpeg, "-hide_banner", "-nostdin", "-i", str(segment),
                 "-frames:v", "1", "-f", "null", "-"],
                capture_output=True, timeout=15,
            )
        except Exception:  # noqa: BLE001
            return None
        m = _DIMS.search(proc.stderr.decode(errors="replace"))
        return (int(m.group(1)), int(m.group(2))) if m else None


def replay_attempts(
    ffmpeg: str,
    source: Path,
    out_pattern: Path,
    seek: float,
    duration: float,
    fps: float,
    crop: tuple[int, int, int, int],
    *,
    hwaccel: bool = False,
) -> list[tuple[str, list[str]]]:
    """The commands to try, in order, to decode `duration` s of ONE segment
    file from `seek` s into it, as ("cuda" | "cpu", argv).

    GPU first when there is one, then the CPU. For each, the seek before the
    input first — ffmpeg jumps to the keyframe before `seek` and decodes
    forward from there, dropping frames until `seek` exactly — then the seek
    after the input, which decodes the file from its start: slower, kept for
    a build that cannot seek this file.
    """
    cw, ch, cx1, cy1 = crop
    common_out = [
        "-t", f"{duration:.3f}", "-an",
        "-vf", f"fps={fps:g},crop={cw}:{ch}:{cx1}:{cy1}",
        "-q:v", "2", str(out_pattern),
    ]
    head = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    attempts: list[tuple[str, list[str]]] = []
    for how, accel in ((("cuda", ["-hwaccel", "cuda"]),) if hwaccel else ()) + (("cpu", []),):
        attempts += [
            (how, head + accel + ["-ss", f"{seek:.3f}", "-i", str(source)] + common_out),
            (how, head + accel + ["-i", str(source), "-ss", f"{seek:.3f}"] + common_out),
        ]
    return attempts


def decode_portions(
    segments: Sequence[tuple[float, Path]], start: float, end: float,
) -> list[tuple[float, Path, float, float]]:
    """Each segment's share of [start, end): (segment start, path, portion
    start, portion end), in order.

    One decode per SEGMENT FILE rather than one over the segments joined with
    the concat demuxer, because seeking a concat input is not exact: measured,
    `-ss` before a concat input landed on the NEXT keyframe — frames labelled
    0.2 s into the window were 1.0 s in, and the window's first 0.8 s was
    missing. For a fast car that is the difference between a plate matched to
    where the car was and one matched to where it had been a second earlier.
    Seeking within one file is exact.
    """
    out = []
    for i, (seg_start, path) in enumerate(segments):
        seg_end = segments[i + 1][0] if i + 1 < len(segments) else end
        p_start, p_end = max(start, seg_start), min(end, seg_end)
        if p_end - p_start > 1e-3:
            out.append((seg_start, path, p_start, p_end))
    return out


#: The original name, kept for existing imports.
PlateReplay = RecordingReplay


def segment_starts(segments: Sequence[tuple[float, Path]]) -> list[tuple[float, Path]]:
    """Segment start times, corrected from the whole-second file names to the
    previous segment's last write where that is consistent (module docstring).

    The recorder cuts a segment the moment the next keyframe arrives, so the
    previous file's mtime IS the new one's start. Trusted only within
    _MTIME_TRUST_S after the named start: across a gap (a recorder restart)
    the previous file is older, and the name is all there is.
    """
    out: list[tuple[float, Path]] = []
    prev_mtime: Optional[float] = None
    for named, path in segments:
        seg_start = named
        if prev_mtime is not None and named <= prev_mtime < named + _MTIME_TRUST_S:
            seg_start = prev_mtime
        out.append((seg_start, path))
        try:
            prev_mtime = path.stat().st_mtime
        except OSError:
            prev_mtime = None
    return out
