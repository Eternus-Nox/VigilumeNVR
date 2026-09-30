"""Read a car's plate again from the RECORDING, after it has gone.

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
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .recorder import build_concat_list, select_segments

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


class PlateReplay:
    """Decodes a vehicle's time window from the recording. One per process."""

    def __init__(
        self,
        camera_dir: Callable[[str], Path],
        ffmpeg: Optional[str] = None,
        *,
        fps: float = REPLAY_FPS,
    ) -> None:
        self._camera_dir = camera_dir
        self._ffmpeg = ffmpeg if ffmpeg is not None else shutil.which("ffmpeg")
        self._fps = fps
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
    ) -> list[ReplayFrame]:
        """Recorded frames for [start, end] (wall-clock epoch), cropped to
        `region` (x1, y1, x2, y2 as FRACTIONS of the frame). [] on anything
        that prevents it. Never raises."""
        if not self._ffmpeg:
            return []
        start = start - WINDOW_PAD_S
        end = min(end + WINDOW_PAD_S, start + MAX_WINDOW_S)
        if end <= start:
            return []
        async with self._lock:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._frames_blocking, camera, start, end, region),
                    DECODE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                log.warning("plate replay on %s took over %.0f s — abandoned",
                            camera, DECODE_TIMEOUT_S)
                return []
            except Exception:  # noqa: BLE001
                log.exception("plate replay on %s failed", camera)
                return []

    # -- blocking work (threads) ------------------------------------------

    def _frames_blocking(
        self, camera: str, start: float, end: float, region: Sequence[float]
    ) -> list[ReplayFrame]:
        import subprocess

        segments = select_segments(self._camera_dir(camera), start, end)
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

        seek = max(0.0, start - segments[0][0])
        with tempfile.TemporaryDirectory(prefix="vigilume-plate-replay-") as tmp:
            listing = Path(tmp) / "list.txt"
            listing.write_text(build_concat_list([p for _, p in segments]))
            common_out = [
                "-t", f"{end - start:.3f}", "-an",
                "-vf", f"fps={self._fps:g},crop={cw}:{ch}:{cx1}:{cy1}",
                "-q:v", "2", str(Path(tmp) / "%04d.jpg"),
            ]
            head = [self._ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
            concat_in = ["-f", "concat", "-safe", "0", "-i", str(listing)]
            attempts = (
                # -ss BEFORE -i: jump to the keyframe before the window instead
                # of decoding the segment from its start (up to 10 s of 4K HEVC
                # for nothing); decoding to images keeps the cut exact.
                head + ["-ss", f"{seek:.3f}"] + concat_in + common_out,
                # -ss AFTER -i: decodes from the segment start, slower, but the
                # form event-clip extraction already relies on. Used only if
                # the fast form produced nothing (seen: a build that crashes
                # seeking before a concat input).
                head + concat_in + ["-ss", f"{seek:.3f}"] + common_out,
            )
            files: list[Path] = []
            for args in attempts:
                for stale in Path(tmp).glob("*.jpg"):
                    stale.unlink()
                proc = subprocess.run(args, capture_output=True, timeout=DECODE_TIMEOUT_S)
                files = sorted(Path(tmp).glob("*.jpg"))
                if files:
                    break
                log.debug("plate replay decode attempt failed on %s (rc %s): %s", camera,
                          proc.returncode, proc.stderr.decode(errors="replace")[-300:])
            if not files:
                return []
            out: list[ReplayFrame] = []
            for i, path in enumerate(files):
                img = cv2.imread(str(path))
                if img is None:
                    continue
                out.append(ReplayFrame(time=start + i / self._fps, crop=img,
                                       ox=cx1, oy=cy1, width=width, height=height))
            return out

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
