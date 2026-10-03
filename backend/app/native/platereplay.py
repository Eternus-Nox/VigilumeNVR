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

#: How long a camera's recorded frame size is trusted before it is probed again.
DIMS_TTL_S = 600.0

#: A decoded crop is scaled down so its longer side is at most this.
MAX_CROP_SIDE = 1920

#: Requests for one camera are served by one decode when their windows overlap
#: or are at most this far apart; the worker waits this long for such a request
#: to arrive before decoding.
COALESCE_GAP_S = 1.0
COALESCE_WAIT_S = 0.06

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


@dataclass
class _Request:
    camera: str
    start: float
    end: float
    region: tuple[float, float, float, float]
    fps: float
    live: bool
    future: "asyncio.Future[list[ReplayFrame]]"


class RecordingReplay:
    """Decodes a time window of a camera's recording. One per process,
    shared by the face and plate passes.

    One decode at a time, across every camera, by a single worker. Requests
    for the same camera and overlapping seconds that are waiting together —
    a face burst and a plate burst for the person getting out of a car — are
    served by ONE decode of their union, each handed its own crop (see
    `_compatible`). Decoding is most of what a burst costs.
    """

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
        #: Requests served, and how many of them shared another's decode.
        self.requests = 0
        self.shared = 0
        # Consecutive replays whose GPU decode produced nothing. A GPU that
        # cannot decode this footage is not asked again after HW_GIVE_UP.
        self._hw_misses = 0
        # Each camera's recorded frame size, and when it was learned — probing
        # it costs an ffmpeg run and a keyframe decode, which used to be paid
        # on every burst (0.58 s of CPU at 4K, a third of the burst itself).
        self._dims: dict[str, tuple[tuple[int, int], float]] = {}
        # camera -> monotonic time until which the size is probed every time
        # (forget_dims, after a resolution change).
        self._probe_until: dict[str, float] = {}
        self._queue: list[_Request] = []
        self._worker: Optional["asyncio.Task[None]"] = None

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
        loop = asyncio.get_running_loop()
        req = _Request(camera, start, end, tuple(float(v) for v in region[:4]),  # type: ignore[arg-type]
                       float(fps) if fps else self._fps, live, loop.create_future())
        self._queue.append(req)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._work(), name="recording-replay")
        try:
            return await req.future
        except asyncio.CancelledError:
            if req in self._queue:
                self._queue.remove(req)
            raise

    async def _work(self) -> None:
        while self._queue:
            # A moment for a request made in the same breath — the face pass
            # and the plate pass handle the same frame one after the other —
            # to join this decode rather than queue for its own.
            await asyncio.sleep(COALESCE_WAIT_S)
            if not self._queue:
                break
            req = self._queue.pop(0)
            if req.future.done():
                continue
            batch = [req] + [r for r in self._queue if _compatible(req, r)]
            for r in batch[1:]:
                self._queue.remove(r)
            try:
                results = await asyncio.wait_for(
                    asyncio.to_thread(self._decode_batch, batch), DECODE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                log.warning("recording read on %s took over %.0f s — abandoned",
                            req.camera, DECODE_TIMEOUT_S)
                results = [[] for _ in batch]
            except Exception:  # noqa: BLE001
                log.exception("recording read on %s failed", req.camera)
                results = [[] for _ in batch]
            self.requests += len(batch)
            self.shared += len(batch) - 1
            for r, res in zip(batch, results):
                if not r.future.done():
                    r.future.set_result(res)

    # -- blocking work (threads) ------------------------------------------

    def _decode_batch(self, batch: list[_Request]) -> list[list[ReplayFrame]]:
        """ONE decode of the batch's union window, each request cut its own
        crop from every frame (see `crops_graph`); each gets its own seconds."""
        start = min(r.start for r in batch)
        end = max(r.end for r in batch)
        per = self._frames_blocking(batch[0].camera, start, end, [r.region for r in batch],
                                    batch[0].fps)
        half = 0.5 / batch[0].fps
        out: list[list[ReplayFrame]] = []
        for r, frames in zip(batch, per):
            if frames and any(q.live for q in batch):
                frames = frames[:-1]  # the last frame of a segment being written can be torn
            out.append([f for f in frames if r.start - half <= f.time <= r.end + half])
        return out

    def forget_dims(self, camera: str, probe_for_s: float = 120.0) -> None:
        """The camera's main stream just changed size (amcrest/stream_profiles).
        For the next `probe_for_s` every window probes its own first segment:
        windows still on old-size segments and new-size ones both come in that
        period, and caching either would mis-place the other's crops."""
        import time as _time

        self._dims.pop(camera, None)
        self._probe_until[camera] = _time.monotonic() + probe_for_s

    def _dims_for(self, camera: str, segment: Path, *, fresh: bool = False) -> Optional[tuple[int, int]]:
        import time as _time

        cached = self._dims.get(camera)
        if _time.monotonic() < self._probe_until.get(camera, 0.0):
            fresh = True
        if cached is not None and not fresh and _time.monotonic() - cached[1] < DIMS_TTL_S:
            return cached[0]
        dims = self._probe_dims(segment)
        if dims is not None:
            self._dims[camera] = (dims, _time.monotonic())
        return dims

    def _frames_blocking(
        self, camera: str, start: float, end: float, regions: Sequence[Sequence[float]],
        fps: Optional[float] = None, *, _retry: bool = True,
    ) -> list[list[ReplayFrame]]:
        """Frames of [start, end] for each region, from ONE decode: a list per
        region. A single region may also be passed bare (old callers)."""
        import subprocess

        if regions and not isinstance(regions[0], (list, tuple)):
            return self._frames_blocking(camera, start, end, [regions], fps, _retry=_retry)[0]  # type: ignore[list-item]
        fps = fps or self._fps
        empty: list[list[ReplayFrame]] = [[] for _ in regions]
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
            return empty
        cached = camera in self._dims
        dims = self._dims_for(camera, segments[0][1])
        if dims is None:
            return empty
        width, height = dims
        crops = [plan_crop(r, width, height) for r in regions]
        usable = [c for c in crops if c is not None]
        if not usable:
            return empty
        stack_w = max(c.ow for c in usable)
        stack_h = sum(c.oh for c in usable)
        frame_bytes = stack_w * stack_h * 3

        out: list[list[ReplayFrame]] = [[] for _ in regions]
        # Each segment's share of the window, decoded from THAT FILE with the
        # seek before its input — see decode_portions for why not one decode
        # over the joined segments. Frames come back as raw BGR on a pipe:
        # writing them out as JPEG files and reading them back cost a fifth of
        # the decode again, for nothing.
        for seg_start, seg_path, p_start, p_end in decode_portions(segments, start, end):
            data = b""
            for how, args in replay_attempts(
                self._ffmpeg, seg_path, p_start - seg_start, p_end - p_start,
                fps, usable, hwaccel=self._hwaccel,
            ):
                proc = subprocess.run(args, capture_output=True, timeout=DECODE_TIMEOUT_S)
                data = proc.stdout or b""
                if len(data) >= frame_bytes:
                    self.decodes[how] += 1
                    if how == "cuda":
                        self._hw_misses = 0
                    elif self._hwaccel:
                        self._note_hw_miss(camera, proc)
                    break
                log.debug("replay decode attempt failed on %s (rc %s): %s", camera,
                          proc.returncode, proc.stderr.decode(errors="replace")[-300:])
            n = len(data) // frame_bytes
            if n == 0:
                continue
            stack = np.frombuffer(data[: n * frame_bytes], np.uint8).reshape(n, stack_h, stack_w, 3)
            row = 0
            for idx, c in enumerate(crops):
                if c is None:
                    continue
                for i in range(n):
                    # Frame i of this portion is i/fps after its start, in the
                    # (possibly scaled) crop's own geometry.
                    out[idx].append(ReplayFrame(
                        time=p_start + i / fps, crop=stack[i, row:row + c.oh, :c.ow],
                        ox=int(round(c.x * c.sx)), oy=int(round(c.y * c.sy)),
                        width=int(round(width * c.sx)), height=int(round(height * c.sy)),
                    ))
                row += c.oh
        if not any(out) and cached and _retry:
            # The camera's frame size may have changed (a camera set from 4K to
            # 1080p): learn it again and try once more.
            self._dims_for(camera, segments[0][1], fresh=True)
            return self._frames_blocking(camera, start, end, regions, fps, _retry=False)
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
                "decodes": dict(self.decodes),
                # Reads asked for, and how many rode along on another's decode.
                "requests": self.requests, "shared": self.shared}

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


@dataclass(frozen=True)
class CropPlan:
    """Where one region is cut from the decoded frame, and what size it comes
    out at (scaled down if its longer side exceeds MAX_CROP_SIDE)."""
    x: int
    y: int
    w: int
    h: int
    ow: int
    oh: int

    @property
    def sx(self) -> float:
        return self.ow / float(self.w)

    @property
    def sy(self) -> float:
        return self.oh / float(self.h)


def plan_crop(region: Sequence[float], width: int, height: int) -> Optional[CropPlan]:
    x1, y1, x2, y2 = _crop_box(region, width, height)
    w, h = (x2 - x1) & ~1, (y2 - y1) & ~1
    if w < 16 or h < 16:
        return None
    # A crop bigger than MAX_CROP_SIDE is scaled down as it is decoded: a car
    # that fills a 4K frame has a plate hundreds of pixels wide, and holding
    # 40 full-size 4K crops would be a gigabyte.
    scale = min(1.0, MAX_CROP_SIDE / float(max(w, h)))
    if scale < 1.0:
        return CropPlan(x1, y1, w, h, max(16, int(w * scale)) & ~1, max(16, int(h * scale)) & ~1)
    return CropPlan(x1, y1, w, h, w, h)


def crops_graph(fps: float, crops: Sequence[CropPlan]) -> str:
    """The filtergraph that cuts every crop from each decoded frame and stacks
    them, top to bottom, into ONE output frame (each padded to the widest) —
    one decode, one pipe, each crop at its own resolution. Split apart again
    by row in `_frames_blocking`."""
    def chain(c: CropPlan) -> str:
        f = f"crop={c.w}:{c.h}:{c.x}:{c.y}"
        return f + (f",scale={c.ow}:{c.oh}" if (c.ow, c.oh) != (c.w, c.h) else "")

    if len(crops) == 1:
        return f"[0:v]fps={fps:g},{chain(crops[0])},format=bgr24[out]"
    width = max(c.ow for c in crops)
    labels = "".join(f"[s{i}]" for i in range(len(crops)))
    parts = [f"[0:v]fps={fps:g},split={len(crops)}{labels}"]
    for i, c in enumerate(crops):
        parts.append(f"[s{i}]{chain(c)},pad={width}:{c.oh}:0:0,format=bgr24[c{i}]")
    parts.append("".join(f"[c{i}]" for i in range(len(crops))) + f"vstack=inputs={len(crops)}[out]")
    return ";".join(parts)


def replay_attempts(
    ffmpeg: str,
    source: Path,
    seek: float,
    duration: float,
    fps: float,
    crops: Any,
    *,
    hwaccel: bool = False,
) -> list[tuple[str, list[str]]]:
    """The commands to try, in order, to decode `duration` s of ONE segment
    file from `seek` s into it, as ("cuda" | "cpu", argv). Every frame's
    crops come out stacked in one raw BGR frame on stdout (`crops_graph`).
    `crops` is a list of CropPlan, or one (w, h, x, y) tuple.

    GPU first when there is one, then the CPU. For each, the seek before the
    input first — ffmpeg jumps to the keyframe before `seek` and decodes
    forward from there, dropping frames until `seek` exactly — then the seek
    after the input, which decodes the file from its start: slower, kept for
    a build that cannot seek this file.
    """
    if isinstance(crops, tuple):
        cw, ch, cx1, cy1 = crops
        crops = [CropPlan(cx1, cy1, cw, ch, cw, ch)]
    common_out = [
        "-t", f"{duration:.3f}", "-an", "-filter_complex", crops_graph(fps, crops),
        "-map", "[out]", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    head = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    attempts: list[tuple[str, list[str]]] = []
    for how, accel in ((("cuda", ["-hwaccel", "cuda"]),) if hwaccel else ()) + (("cpu", []),):
        attempts += [
            (how, head + accel + ["-ss", f"{seek:.3f}", "-i", str(source)] + common_out),
            (how, head + accel + ["-i", str(source), "-ss", f"{seek:.3f}"] + common_out),
        ]
    return attempts


def _crop_box(region: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """A fractional region, plus REGION_MARGIN, in a `width` x `height`
    frame's pixels: (x1, y1, x2, y2), x1/y1 even."""
    x1, y1, x2, y2 = (float(v) for v in region[:4])
    mx, my = (x2 - x1) * REGION_MARGIN, (y2 - y1) * REGION_MARGIN
    return (max(0, int((x1 - mx) * width)) & ~1, max(0, int((y1 - my) * height)) & ~1,
            min(width, int(round((x2 + mx) * width))), min(height, int(round((y2 + my) * height))))


def _compatible(a: _Request, b: _Request) -> bool:
    """Can `b` share `a`'s decode? Same camera and rate, seconds that overlap
    or nearly meet, and a union window no longer than one read may be. Where
    in the frame each one is does not matter: the decoder decodes the whole
    frame either way, and each gets its own crop of it."""
    if a.camera != b.camera or a.fps != b.fps:
        return False
    if b.start > a.end + COALESCE_GAP_S or a.start > b.end + COALESCE_GAP_S:
        return False
    return max(a.end, b.end) - min(a.start, b.start) <= MAX_WINDOW_S


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
