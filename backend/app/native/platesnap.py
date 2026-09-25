"""Full-resolution frames for the plate pass.

WHY THIS EXISTS
===============
Detection decodes each camera's SUBSTREAM, scaled to the detect size — 704x480
by default. That is plenty to find a car, and far too little to read its plate:
a car filling a third of the frame carries a plate about 40 px wide, and the
plate scorer needs 64 px before a crop is worth reading at all (bestshot.
PLATE_MIN_PX; ~160 px for a clean read). Measured on a synthetic driveway, the
substream produced no plate regions whatsoever until the car filled more than
half the frame. Plates were only being read when a car was right up at the lens.

The camera already has the pixels — its main stream is three to six times
wider. Decoding that continuously for every camera would cost more than
detection itself, so this asks the camera for ONE full-resolution JPEG
(`snapshot.cgi`) only while a vehicle is being tracked on a camera with plate
reading on, at most about once a second per vehicle, and stops once the plate
has been read confidently.

LINING THE TWO FRAMES UP
------------------------
The snapshot is taken AFTER the detect frame — by however long ingest,
detection and the HTTP round trip took, a few hundred milliseconds in which a
moving car moves. So the detect-frame box is not simply scaled up. The
snapshot is shrunk to the detect frame's size and the vehicle found in it by
template matching against the detect-frame crop, near where it was; only then
is the box scaled up and the plate cut from the full-resolution pixels. A
match below `MATCH_MIN` means the car moved too far or the view differs, and
that look is skipped rather than guessed at: a plate cut from the wrong place
is a wrong plate.

Both frames are the same sensor's full field of view, so one per-axis scale
maps between them even when the substream is anamorphic (704x480 is 4:3-ish;
the main stream is usually 16:9).

NEVER RAISES INTO THE CALLER
----------------------------
A camera that is not an Amcrest/Dahua, has no IP, rejects the credentials or
is simply slow is a normal state: the fetch returns None, the failure is
recorded for the status screen, and after a few in a row the camera is left
alone for a while instead of being asked again every second.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: A snapshot slower than this is too late to be the same moment. The fetch is
#: abandoned rather than waited for.
FETCH_TIMEOUT_S = 2.5

#: Consecutive failures before a camera is left alone for `BACKOFF_S`.
FAIL_LIMIT = 3
BACKOFF_S = 300.0

#: One snapshot serves every vehicle on the camera that asks within this long,
#: so two cars on a drive cost one request, not two.
SHARE_S = 0.35

#: A full-resolution frame must be at least this much wider than the detect
#: frame to be worth fetching. Some cameras serve snapshots at the substream's
#: size; asking for those every second buys nothing.
MIN_GAIN = 1.3

#: How far beyond the detect-frame box the vehicle is searched for, as a
#: fraction of the box's size on each side. Generous, because the snapshot
#: arrives late; bounded, because a whole-frame search could lock onto another
#: parked car.
SEARCH_MARGIN = 0.75

#: Template scales tried: the car may have come closer or moved away.
MATCH_SCALES = (1.0, 0.9, 1.1)

#: Normalized cross-correlation below which the match is not trusted.
MATCH_MIN = 0.5

#: A detect-frame vehicle smaller than this cannot be matched reliably — and
#: is too far away for its plate to be legible even at full resolution.
MIN_TEMPLATE_PX = 24


class SnapshotUnavailable(RuntimeError):
    """The camera cannot be asked for a snapshot at all (e.g. no IP)."""


@dataclass
class _Health:
    ok: int = 0
    failed: int = 0
    consecutive: int = 0
    last_error: str = ""
    resolution: str = ""
    latency_ms: float = 0.0
    retry_at: float = 0.0
    #: Set when the camera answers, but with nothing more than detection has.
    no_gain: str = ""

    def note_success(self, shape: Sequence[int], seconds: float) -> None:
        self.ok += 1
        self.consecutive = 0
        self.last_error = ""
        self.resolution = f"{shape[1]}x{shape[0]}"
        ms = seconds * 1000.0
        # A slow EWMA so one hiccup does not dominate the status screen.
        self.latency_ms = ms if self.ok == 1 else self.latency_ms * 0.8 + ms * 0.2

    def note_failure(self, reason: str) -> None:
        self.failed += 1
        self.consecutive += 1
        self.last_error = reason
        if self.consecutive >= FAIL_LIMIT:
            self.retry_at = time.monotonic() + BACKOFF_S
            self.consecutive = 0

    def report(self) -> dict[str, Any]:
        wait = max(0.0, self.retry_at - time.monotonic())
        return {
            "ok": self.ok,
            "failed": self.failed,
            "last_error": self.last_error,
            "resolution": self.resolution,
            "latency_ms": round(self.latency_ms, 1),
            "backing_off_s": round(wait) if wait else 0,
            "no_gain": self.no_gain,
        }


@dataclass
class _Client:
    key: tuple[str, str, str]
    client: Any


FetchJpeg = Callable[[dict[str, Any]], Awaitable[bytes]]


@dataclass
class SnapshotSource:
    """Full-resolution JPEGs from the cameras, shared, rate-limited, never raising.

    `fetch_jpeg` is injectable so tests can serve frames without a camera; the
    default asks the camera's own `snapshot.cgi` (Amcrest/Dahua).
    """

    fetch_jpeg: Optional[FetchJpeg] = None
    _health: dict[str, _Health] = field(default_factory=dict)
    _inflight: dict[str, "asyncio.Future[Optional[tuple[np.ndarray, float]]]"] = field(
        default_factory=dict
    )
    _recent: dict[str, tuple[float, np.ndarray, float]] = field(default_factory=dict)
    _clients: dict[str, _Client] = field(default_factory=dict)

    # ---------- asking ----------

    def available(self, camera: str) -> bool:
        """Whether it is worth asking this camera right now."""
        h = self._health.get(camera)
        if h is None:
            return True
        if h.no_gain:
            return False
        return h.retry_at <= time.monotonic()

    async def fetch(self, cam_row: dict[str, Any]) -> Optional[tuple[np.ndarray, float]]:
        """``(frame_bgr, wall_time_taken)``, or None. Never raises."""
        camera = str(cam_row.get("name") or "")
        if not camera or not self.available(camera):
            return None
        recent = self._recent.get(camera)
        if recent is not None and time.monotonic() - recent[0] <= SHARE_S:
            return recent[1], recent[2]
        fut = self._inflight.get(camera)
        if fut is None or fut.done():
            fut = asyncio.ensure_future(self._fetch_one(camera, dict(cam_row)))
            self._inflight[camera] = fut
        try:
            # Shielded: one caller being cancelled (its track ended) must not
            # cancel the request another vehicle on the same camera is sharing.
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — _fetch_one already swallows; belt and braces
            return None

    async def _fetch_one(
        self, camera: str, cam_row: dict[str, Any]
    ) -> Optional[tuple[np.ndarray, float]]:
        h = self._health.setdefault(camera, _Health())
        fetch = self.fetch_jpeg or self._amcrest_jpeg
        started = time.monotonic()
        try:
            data = await asyncio.wait_for(fetch(cam_row), FETCH_TIMEOUT_S)
            frame = await asyncio.to_thread(_decode, data)
            if frame is None:
                raise ValueError("the camera's snapshot did not decode as a JPEG")
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            h.note_failure(f"the camera took longer than {FETCH_TIMEOUT_S:g} s to send a snapshot")
            return None
        except Exception as exc:  # noqa: BLE001
            h.note_failure(_describe(exc))
            log.debug("plate snapshot from %s failed: %s", camera, exc)
            return None
        taken = time.time()
        h.note_success(frame.shape, time.monotonic() - started)
        stamp = time.monotonic()
        self._recent[camera] = (stamp, frame, taken)
        # Dropped once it can no longer be shared: a full-resolution frame is
        # several megabytes, and holding one per camera between vehicles would
        # be memory spent on nothing.
        asyncio.get_running_loop().call_later(
            SHARE_S * 2, self._drop_recent, camera, stamp
        )
        return frame, taken

    def _drop_recent(self, camera: str, stamp: float) -> None:
        recent = self._recent.get(camera)
        if recent is not None and recent[0] == stamp:
            del self._recent[camera]

    def note_no_gain(self, camera: str, hires_shape: Sequence[int],
                     detect_shape: Sequence[int]) -> None:
        """The camera's snapshot is no bigger than the detect frame: stop asking.

        Recorded as a standing reason rather than a failure, because nothing
        failed — the fix is a camera setting, and the status screen says which.
        """
        h = self._health.setdefault(camera, _Health())
        h.no_gain = (
            f"the camera's snapshot is only {hires_shape[1]}x{hires_shape[0]}, no more "
            f"detail than detection's {detect_shape[1]}x{detect_shape[0]}. Raise the "
            "snapshot resolution in the camera's encode settings."
        )

    async def _amcrest_jpeg(self, cam_row: dict[str, Any]) -> bytes:
        from ..amcrest.client import AmcrestClient

        camera = str(cam_row.get("name") or "")
        ip = str(cam_row.get("ip") or "").strip()
        if not ip:
            raise SnapshotUnavailable("the camera has no IP address to ask for a snapshot")
        key = (ip, str(cam_row.get("username") or ""), str(cam_row.get("password") or ""))
        entry = self._clients.get(camera)
        if entry is None or entry.key != key:
            if entry is not None:
                await _close_quietly(entry.client)
            # Kept open between requests: Digest auth reuses its challenge, so
            # the second snapshot costs one round trip instead of two.
            entry = _Client(key, AmcrestClient(ip, key[1], key[2], timeout=FETCH_TIMEOUT_S))
            self._clients[camera] = entry
        return await entry.client.snapshot()

    # ---------- lifetime ----------

    def forget(self, camera: str) -> None:
        self._recent.pop(camera, None)
        self._health.pop(camera, None)
        entry = self._clients.pop(camera, None)
        if entry is not None:
            asyncio.ensure_future(_close_quietly(entry.client))

    async def close(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        self._recent.clear()
        for entry in clients:
            await _close_quietly(entry.client)

    def status(self) -> dict[str, Any]:
        return {camera: h.report() for camera, h in sorted(self._health.items())}


def locate(
    template_bgr: np.ndarray,
    box: Sequence[float],
    detect_shape: Sequence[int],
    hires_bgr: np.ndarray,
) -> Optional[tuple[tuple[float, float, float, float], float]]:
    """Find the vehicle in a full-resolution frame.

    ``template_bgr`` is the vehicle as the DETECT frame saw it, cut at ``box``
    (detect-frame pixels); ``detect_shape`` is that frame's ``(h, w)``. Returns
    ``(box_in_hires_pixels, match_score)``, or None when the vehicle cannot be
    found with confidence — in which case the caller skips this look.
    """
    if template_bgr is None or template_bgr.size == 0 or hires_bgr is None or hires_bgr.size == 0:
        return None
    dh, dw = int(detect_shape[0]), int(detect_shape[1])
    hh, hw = hires_bgr.shape[:2]
    if dw < 2 or dh < 2:
        return None
    tmpl = _gray(template_bgr)
    th, tw = tmpl.shape[:2]
    if tw < MIN_TEMPLATE_PX or th < MIN_TEMPLATE_PX:
        return None
    # A featureless template matches everywhere equally, which is the same as
    # matching nowhere.
    if float(tmpl.std()) < 4.0:
        return None

    small = _gray(cv2.resize(hires_bgr, (dw, dh), interpolation=cv2.INTER_AREA))
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    mx, my = (x2 - x1) * SEARCH_MARGIN, (y2 - y1) * SEARCH_MARGIN
    wx1, wy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    wx2, wy2 = min(dw, int(round(x2 + mx))), min(dh, int(round(y2 + my)))
    window = small[wy1:wy2, wx1:wx2]

    best: Optional[tuple[float, int, int, int, int]] = None
    for scale in MATCH_SCALES:
        if scale == 1.0:
            t = tmpl
        else:
            sw, sh = int(round(tw * scale)), int(round(th * scale))
            if sw < MIN_TEMPLATE_PX or sh < MIN_TEMPLATE_PX:
                continue
            t = cv2.resize(tmpl, (sw, sh), interpolation=cv2.INTER_AREA)
        if window.shape[0] < t.shape[0] or window.shape[1] < t.shape[1]:
            continue
        result = cv2.matchTemplate(window, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if not np.isfinite(score):
            continue
        if best is None or score > best[0]:
            best = (float(score), wx1 + loc[0], wy1 + loc[1], t.shape[1], t.shape[0])

    if best is None or best[0] < MATCH_MIN:
        return None
    score, bx, by, bw, bh = best
    sx, sy = hw / float(dw), hh / float(dh)
    return (bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy), score


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _decode(data: bytes) -> Optional[np.ndarray]:
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _describe(exc: BaseException) -> str:
    """A reason a person can act on, for the status screen."""
    text = str(exc) or type(exc).__name__
    lowered = text.lower()
    if "401" in lowered or "unauthorized" in lowered:
        return "the camera rejected the stored username/password for snapshots"
    if "404" in lowered or "not found" in lowered:
        return "the camera has no snapshot.cgi — full-resolution plate reading needs an Amcrest/Dahua camera"
    return text[:200]


async def _close_quietly(client: Any) -> None:
    try:
        await client.aclose()
    except Exception:  # noqa: BLE001
        pass
