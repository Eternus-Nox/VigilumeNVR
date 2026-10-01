"""The per-track plate pass: watch a vehicle, read its plate several times,
and let the reads outvote each other.

MIRRORS facepass.py, WITH ONE STRUCTURAL DIFFERENCE
===================================================
A face produces ONE embedding that is matched once. A plate produces a STRING
per look, and the whole accuracy argument for this feature is that several
independent looks beat one good one (see recognition.vote_plate). So where the
face pass embeds once per track and re-embeds only on a much better shot, this
pass OCRs every shot it retains — at most `KEEP_SHOTS` of them — and votes.

That is affordable because the pinned OCR is 3.3 MB over a 128x64 input, so
five reads per vehicle is nothing next to the D-FINE pass that found the car.

WHY THE READS ARE INDEPENDENT (AND WHY THAT MATTERS)
----------------------------------------------------
Running several OCR networks over ONE frame mostly buys correlated errors —
they all fail on the same motion blur. Running one OCR over shots taken at
different instants, which `bestshot.BestShotBuffer` already spreads out in
time, gives errors that are genuinely independent. A per-character weighted
vote then recovers the true string from reads where no single frame got it
entirely right. This is the honest version of "several small models looking
for the same indicator".

FINDING THE PLATE
-----------------
A learned plate detector (see plates.py) finds the plate in the vehicle crop,
or directly in a full-resolution snapshot around the vehicle. When it is not
loaded — disabled by `recognition.plate_detector`, or not downloadable — the
classical localizer (`plates.candidate_regions`) proposes strips instead, and
the readers reject the ones that were never plates. Every retained crop is read
by every loaded reader, and each read is a separate vote.

TWO SOURCES OF PIXELS
---------------------
Every pass looks at the DETECT frame it was handed, as it always has. That
frame is the camera's substream scaled to 704x480 or so, where a plate is
usually too small to read — so, on top, a vehicle being tracked also triggers a
FULL-RESOLUTION look (`platesnap`): a snapshot from the camera, the vehicle
found in it, the plate cut from the real pixels. That runs as a background task
because an HTTP round trip must never stall the detection loop; its reads land
in the same buffer and the same vote, and they carry more weight there simply
because a sharper, wider crop scores higher.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import platesnap
from . import zones as zonelib
from .bestshot import (
    KEEP_SHOTS, MIN_GAP_S, BestShotBuffer, Shot, encode_frame_box, score_plate,
    shot_params,
)
from .heatmap import HeatmapAccumulator
from .plates import OCR_MIN_CONFIDENCE, PlateReader, deskew, is_vehicle
from .platereplay import PlateReplay
from .platesnap import SnapshotSource
from .recognition import (
    PLATE_REGIONS, Gallery, Match, PlateRead, PlateVote, regional_plate,
    vote_plate,
)
from .recognizer import crop_with_origin

log = logging.getLogger(__name__)

#: Minimum spacing between plate passes on ONE track.
PASS_INTERVAL_S = 0.7

#: Padding around the vehicle box before looking for a plate. Smaller than the
#: face pass's: a vehicle box is large and already contains its plate, so the
#: padding is only to survive a box that clips the bumper.
VEHICLE_CROP_PAD = 0.04

#: A vote below this is not worth recording. Plate voting reports its WEAKEST
#: character as the confidence, so this means "no character was a coin flip".
#:
#: 0.7, raised from 0.6 on measurement: on 222 real US plates, single frame,
#: 0.6 stored 203 right plates and 9 wrong ones; 0.7 stored 200 right and 5
#: wrong. A wrong plate on an event is worse than none, and more frames of the
#: same car push a right answer's confidence up, not a wrong one's.
MIN_VOTE_CONFIDENCE = 0.7

#: Plates shorter than this are almost always a partial read of a longer one.
MIN_PLATE_LENGTH = 4

MAX_CANDIDATES = 2000
RUN_INTERVAL_S = 300.0
DEFAULT_RETENTION_DAYS = 7.0

#: Minimum spacing between two FULL-RESOLUTION looks at one vehicle. Each is an
#: HTTP request to the camera, so this is set by what the camera will happily
#: serve, not by what the OCR could absorb.
HIRES_INTERVAL_S = 1.0

#: INSIDE A PLATE ZONE the pass runs flat out. The zone is the operator
#: saying "plates are readable here — read them", and a car driving through a
#: small box is in it for a second or two: at the ordinary cadence it could be
#: gone before the second full-resolution look. So in a zone the detect frame
#: is searched every frame (at 5 fps), a full-resolution look goes out every
#: half second, and — see `observe` — the car is read from the first frame it
#: is tracked, without waiting for it to be confirmed as a moving subject.
ZONE_PASS_INTERVAL_S = 0.15

#: Margins (fraction of the vehicle box) tried in turn when searching a
#: full-resolution snapshot for the plate: tight first, wider only if nothing
#: was found. Wider when the car could not be matched in the snapshot, since
#: then only its detect-frame position is known and it has moved since.
HIRES_MARGINS_FOUND = (0.05, 0.25)
HIRES_MARGINS_LOST = (0.1, 0.4, 0.8)
ZONE_HIRES_INTERVAL_S = 0.5

#: Most full-resolution looks one vehicle gets. A car that parks and stays
#: tracked must not cost a snapshot a second for as long as it sits there.
HIRES_MAX_PER_TRACK = 8

#: A vote this confident from this many reads is settled — further looks would
#: only confirm it, so they are not taken.
SETTLED_CONFIDENCE = 0.9
SETTLED_READS = 3

#: How long a vehicle's final vote waits for a full-resolution look that was
#: still in flight when the track ended. The wait happens in the background;
#: the detection loop never waits on it.
HIRES_FINISH_WAIT_S = 3.0

#: A replayed plate crop scoring below this is not read. The same floor the
#: best-shot buffer applies, so a replay cannot add votes from crops the live
#: path would have thrown away.
MIN_REPLAY_QUALITY = 0.25


@dataclass
class _CameraStats:
    """Where plates are being lost, per camera, since the backend started.

    Every counter is a stage a plate has to get through, so the first one that
    stays at zero while the one before it climbs is the answer to "why is it
    not reading plates" — and each has a different remedy.
    """

    passes: int = 0
    regions: int = 0
    too_small: int = 0
    reads: int = 0
    rejected_reads: int = 0
    hires_requested: int = 0
    hires_frames: int = 0
    hires_lost: int = 0
    hires_reads: int = 0
    replays: int = 0
    replay_frames: int = 0
    replay_reads: int = 0
    votes_stored: int = 0
    votes_discarded: int = 0
    last_plate: str = ""
    last_plate_at: float = 0.0
    #: Widths of recent plate-shaped strips from the detect frame, for the
    #: "how far off is it" hint.
    widths: deque = field(default_factory=lambda: deque(maxlen=50))

    def report(self) -> dict[str, Any]:
        return {
            "passes": self.passes,
            "regions": self.regions,
            "too_small": self.too_small,
            "reads": self.reads,
            "rejected_reads": self.rejected_reads,
            "hires_requested": self.hires_requested,
            "hires_frames": self.hires_frames,
            "hires_lost": self.hires_lost,
            "hires_reads": self.hires_reads,
            "replays": self.replays,
            "replay_frames": self.replay_frames,
            "replay_reads": self.replay_reads,
            "votes_stored": self.votes_stored,
            "votes_discarded": self.votes_discarded,
            "last_plate": self.last_plate,
            "last_plate_at": self.last_plate_at,
            "median_strip_px": int(statistics.median(self.widths)) if self.widths else 0,
        }


@dataclass
class _TrackState:
    buffer: BestShotBuffer = field(default_factory=BestShotBuffer)
    last_pass: float = 0.0
    camera: str = ""
    #: One OCR read per retained shot. The vote runs over these.
    reads: list[PlateRead] = field(default_factory=list)
    #: Shots already OCR'd, keyed by (frame_time, box).
    #:
    #: NOT by frame_time alone. Several candidate regions are proposed per
    #: frame and they all carry the SAME frame_time, so the best-shot buffer's
    #: time-diversity rule makes them compete for one slot — a better region
    #: REPLACES a worse one from the same frame. Keyed by time alone, that
    #: replacement would be skipped as "already read" and the frame's best plate
    #: crop would never reach the OCR at all.
    read_shots: set[tuple[float, tuple[float, float, float, float]]] = field(
        default_factory=set
    )
    vote: Optional[PlateVote] = None
    match: Optional[Match] = None
    event_fid: str = ""
    #: The full-resolution look in flight for this vehicle, if any.
    hires_task: Optional[asyncio.Task] = None
    hires_last: float = float("-inf")
    hires_requests: int = 0
    #: Shots that came from a full-resolution frame, keyed like `read_shots`,
    #: so the status screen can say which source the reads are coming from.
    hires_keys: set[tuple[float, tuple[float, float, float, float]]] = field(
        default_factory=set
    )
    #: Set once the vote has been stored (or discarded). A full-resolution look
    #: that lands after that must not announce a different answer.
    concluded: bool = False
    #: Where the vehicle was, over time: (wall time, box as FRACTIONS of the
    #: frame). What the recording replay needs to know which seconds to decode
    #: and which plate in each frame is this vehicle's.
    path: deque = field(default_factory=lambda: deque(maxlen=600))


def _expand(
    box: Sequence[float], frac: float, w: int, h: int
) -> tuple[int, int, int, int]:
    """`box` grown by `frac` of its size on every side, clamped to w x h."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)),
            min(w, int(round(x2 + dx))), min(h, int(round(y2 + dy))))


class PlatePass:
    """Per-camera, per-track licence-plate reading. One instance per engine."""

    def __init__(
        self,
        reader: PlateReader,
        db: Any,
        images_dir: Path,
        heatmap: Optional[HeatmapAccumulator] = None,
        on_recognition: Optional[Any] = None,
        snapshots: Optional[SnapshotSource] = None,
        replay: Optional[PlateReplay] = None,
    ) -> None:
        # See FacePass.on_recognition — the live hook, not the stored row.
        self.on_recognition = on_recognition
        self._reader = reader
        self._db = db
        self._images_dir = Path(images_dir)
        self._tracks: dict[tuple[str, int], _TrackState] = {}
        # Shot-buffer shape, refreshed from settings in run(). More shots
        # matter MORE for plates than for faces: the pass OCRs every
        # retained shot and votes per character, so each extra distinct
        # look is another independent vote against a misread.
        self._shots = KEEP_SHOTS
        self._shot_gap = MIN_GAP_S
        # Shared with the face pass when both are running: one accumulator, two
        # kinds. Its own instance when constructed standalone (tests).
        self.heatmap = heatmap if heatmap is not None else HeatmapAccumulator(db)
        self._gallery: Optional[Gallery] = None
        self._gallery_lock = asyncio.Lock()
        # Full-resolution looks. None disables them outright (tests that want
        # the detect-frame path alone); the setting switches them at runtime.
        self._snapshots = snapshots
        # Reading the recording after a vehicle leaves (platereplay.py). None
        # disables it (tests that want the live path alone).
        self._replay = replay
        self._settings: Any = None
        self._stats: dict[str, _CameraStats] = {}
        # Final votes deferred behind an in-flight full-resolution look, held
        # so they are not garbage-collected mid-flight.
        self._finishing: set[asyncio.Task] = set()

    def _stats_for(self, camera: str) -> _CameraStats:
        st = self._stats.get(camera)
        if st is None:
            st = self._stats[camera] = _CameraStats()
        return st

    def _sync_detector_setting(self) -> None:
        """Apply `recognition.plate_detector` live (default on)."""
        if self._settings is None:
            return
        try:
            cfg = (self._settings.current or {}).get("recognition") or {}
            self._reader.use_plate_detector = bool(cfg.get("plate_detector", True))
        except Exception:  # noqa: BLE001
            pass

    def _plate_region(self) -> str:
        """`recognition.plate_region`, read live. "us" by default."""
        if self._settings is None:
            return "us"
        try:
            cfg = (self._settings.current or {}).get("recognition") or {}
        except Exception:  # noqa: BLE001
            return "us"
        region = str(cfg.get("plate_region") or "us").lower()
        return region if region in PLATE_REGIONS else "us"

    def _hires_on(self) -> bool:
        """Whether full-resolution looks are wanted. Read live, not on the
        five-minute maintenance tick, so the switch takes effect at once."""
        if self._snapshots is None:
            return False
        if self._settings is None:
            return True
        try:
            cfg = (self._settings.current or {}).get("recognition") or {}
        except Exception:  # noqa: BLE001
            return True
        return bool(cfg.get("plate_hires", True))

    # ---------- gallery ----------

    async def reload_gallery(self) -> None:
        """Rebuild the vehicle gallery. Plates carry no embedding, so unlike the
        face gallery this one is unaffected by which model is loaded."""
        async with self._gallery_lock:
            try:
                profiles = await (
                    await self._db.conn.execute("SELECT * FROM profiles WHERE kind = 'vehicle'")
                ).fetchall()
                samples = await (
                    await self._db.conn.execute(
                        "SELECT * FROM profile_samples WHERE plate != ''"
                    )
                ).fetchall()
            except Exception:
                log.exception("could not load the vehicle gallery")
                return
            self._gallery = Gallery.build(
                [dict(r) for r in profiles], [dict(r) for r in samples], model_key=""
            )
            log.info("vehicle gallery loaded: %d profile(s)", len(self._gallery))

    # ---------- per-frame ----------

    async def observe(
        self,
        cam: Any,
        observations: Sequence[Any],
        frame_bgr: Optional[np.ndarray],
        frame_time: float,
        event_fid: str = "",
        seen: Optional[Sequence[Any]] = None,
    ) -> None:
        """Look for a plate on the vehicles in this frame. Never raises.

        `observations` are the CONFIRMED, moving objects — what the ordinary
        pass reads. `seen` is every tracked object this frame, before
        confirmation and before the stationary filter; on a camera with a
        plate zone, any vehicle in it is read from `seen`, at the zone cadence.
        That filter and the three-frame confirmation exist to decide what
        deserves an EVENT, and cost half a second or more; for a plate the
        only question is whether there is a readable one in the box, and the
        OCR already rejects anything that is not a plate.
        """
        if frame_bgr is None or not self._reader.ready:
            return
        self._sync_detector_setting()
        # Per-camera opt-out, checked before any work — see the twin in
        # facepass. A camera that never sees a plate at a readable angle costs
        # localization on every vehicle and returns only marginal crops, which
        # is exactly where a WRONG PLATE comes from.
        if not getattr(cam, "plate_recognition", True):
            return
        camera = cam.row.get("name", "")
        try:
            plate_zones = getattr(cam, "plate_zones", None)
            pool = seen if (plate_zones and seen is not None) else observations
            vehicles = [o for o in pool if is_vehicle(o.label)]
            if not vehicles:
                return
            if plate_zones:
                hits = zonelib.zone_hits(vehicles, plate_zones)
                vehicles = [o for o, names in zip(vehicles, hits) if names]
                if not vehicles:
                    return
            fast = bool(plate_zones)
            for obs in vehicles:
                await self._observe_one(cam, camera, obs, frame_bgr, frame_time, event_fid,
                                        fast=fast)
        except Exception:
            log.exception("plate pass failed on %s", camera)

    async def _observe_one(
        self, cam: Any, camera: str, obs: Any, frame_bgr: np.ndarray,
        frame_time: float, event_fid: str, *, fast: bool = False,
    ) -> None:
        key = (camera, obs.tracker_id)
        st = self._tracks.get(key)
        if st is None:
            st = _TrackState(
                camera=camera,
                buffer=BestShotBuffer(keep=self._shots, min_gap_s=self._shot_gap),
            )
            self._tracks[key] = st
        if event_fid:
            st.event_fid = event_fid
        # Every sighting, before the throttle: the replay needs the whole path.
        fh0, fw0 = frame_bgr.shape[:2]
        if fw0 > 0 and fh0 > 0:
            x1, y1, x2, y2 = (float(v) for v in obs.box[:4])
            st.path.append((frame_time, (x1 / fw0, y1 / fh0, x2 / fw0, y2 / fh0)))
        if frame_time - st.last_pass < (ZONE_PASS_INTERVAL_S if fast else PASS_INTERVAL_S):
            return
        st.last_pass = frame_time
        self._stats_for(camera).passes += 1

        # Before the detect-frame work, so the snapshot request is on the wire
        # while this frame is being searched.
        self._maybe_look_hires(cam, camera, st, obs, frame_bgr, frame_time, fast=fast)

        # Once this camera has PROVEN it serves full-resolution snapshots, the
        # detect frame is not read at all. Measured on 222 real US plates run
        # through this pass: 846 of 1,425 reads came from the detect frame,
        # where the plate is a few dozen pixels, and they were most of why a
        # vote came out wrong or too unsure to store — a blurry read of the
        # same plate disagrees with a sharp one and dilutes it. Until a
        # snapshot has worked (or when they stop working), the detect frame is
        # all there is and it is read as before.
        if self._hires_on() and self._snapshots is not None and self._snapshots.proven(camera):
            return

        cropped = crop_with_origin(frame_bgr, obs.box, pad=VEHICLE_CROP_PAD)
        if cropped is None:
            return
        vehicle, ox, oy = cropped

        fh, fw = frame_bgr.shape[:2]
        await self._offer_regions(
            st, camera, obs.tracker_id, vehicle, ox, oy, fw, fh, frame_time, hires=False
        )

        # OCR whatever the buffer ACTUALLY kept, once the frame's regions have
        # finished competing for its slots. Reading inside the loop above would
        # read crops that a later, better region from the same frame then
        # displaced.
        await self._read_pending(st, obs.tracker_id)

        # Vote as soon as there is something to vote on, so a notification can
        # name the vehicle while it is still on the drive.
        if len(st.reads) >= 2:
            self._tally(st)

    async def _offer_regions(
        self, st: _TrackState, camera: str, tracker_id: int, vehicle: np.ndarray,
        ox: int, oy: int, fw: int, fh: int, frame_time: float, *, hires: bool,
        regions: Optional[Sequence[tuple[int, int, int, int]]] = None,
    ) -> None:
        """Localize plate strips in one vehicle crop and offer them to the buffer.

        `fw`/`fh` are the dimensions of the frame the crop was cut from, which
        is what makes the stored `frame_box` and the heatmap point normalized —
        and so comparable between a detect frame and a full-resolution one.
        """
        stats = self._stats_for(camera)
        if regions is None:
            regions = await asyncio.to_thread(self._reader.find_plates, vehicle)
        if not regions:
            return

        for (x1, y1, x2, y2) in regions:
            strip = vehicle[y1:y2, x1:x2]
            if strip.size == 0:
                continue
            straight = await asyncio.to_thread(deskew, strip)
            quality = score_plate(straight)
            if quality.resolution <= 0.0:
                stats.too_small += 1
            else:
                stats.regions += 1
            if not hires:
                stats.widths.append(x2 - x1)
            # Where this strip sits in the FULL frame, for the review UI to
            # ring. Computed before offering because `shot.box` stays in the
            # vehicle crop's coordinates — see the heatmap note below for the
            # same translation and the same trap.
            frame_box = (
                (
                    max(0.0, min(1.0, (ox + x1) / fw)),
                    max(0.0, min(1.0, (oy + y1) / fh)),
                    max(0.0, min(1.0, (ox + x2) / fw)),
                    max(0.0, min(1.0, (oy + y2) / fh)),
                )
                if fw > 0 and fh > 0
                else None
            )
            shot = st.buffer.offer(
                tracker_id=tracker_id, kind="plate", crop_bgr=straight,
                box=(x1, y1, x2, y2), frame_time=frame_time, quality=quality,
                frame_box=frame_box,
            )
            if shot is None:
                continue
            if hires:
                st.hires_keys.add((shot.frame_time, shot.box))

            # Heatmap: the region is in the VEHICLE CROP's coordinates, so it
            # must be translated by the crop origin before it means anything on
            # the frame. Same trap as the face pass — a missing translation
            # paints a plausible map of the wrong places.
            if fw > 0 and fh > 0:
                cx = (ox + (x1 + x2) / 2.0) / fw
                cy = (oy + (y1 + y2) / 2.0) / fh
                self.heatmap.record(camera, "plate", cx, cy, quality.total)

    # ---------- full-resolution looks ----------

    def _maybe_look_hires(
        self, cam: Any, camera: str, st: _TrackState, obs: Any,
        frame_bgr: np.ndarray, frame_time: float, *, fast: bool = False,
    ) -> None:
        """Start a full-resolution look at this vehicle if one is due. Never awaits."""
        if not self._hires_on():
            return
        assert self._snapshots is not None
        if st.hires_task is not None and not st.hires_task.done():
            return
        if st.hires_requests >= HIRES_MAX_PER_TRACK:
            return
        if frame_time - st.hires_last < (ZONE_HIRES_INTERVAL_S if fast else HIRES_INTERVAL_S):
            return
        if (st.vote is not None and st.vote.reads >= SETTLED_READS
                and st.vote.confidence >= SETTLED_CONFIDENCE):
            return
        if not self._snapshots.available(camera):
            return
        fh, fw = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in obs.box[:4])
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(fw, int(round(x2))), min(fh, int(round(y2)))
        if xi2 - xi1 < platesnap.MIN_TEMPLATE_PX or yi2 - yi1 < platesnap.MIN_TEMPLATE_PX:
            return
        # A copy of the vehicle only, not the whole frame: the engine reuses
        # its frames, and this is all the matcher needs.
        template = frame_bgr[yi1:yi2, xi1:xi2].copy()
        st.hires_last = frame_time
        st.hires_requests += 1
        self._stats_for(camera).hires_requested += 1
        row = dict(getattr(cam, "row", None) or {})
        row.setdefault("name", camera)
        st.hires_task = asyncio.create_task(
            self._look_hires(row, camera, st, obs.tracker_id, template,
                             (xi1, yi1, xi2, yi2), (fh, fw)),
            name=f"plate-hires-{camera}-{obs.tracker_id}",
        )

    async def _look_hires(
        self, cam_row: dict[str, Any], camera: str, st: _TrackState, tracker_id: int,
        template: np.ndarray, box: tuple[int, int, int, int],
        detect_shape: tuple[int, int],
    ) -> None:
        """Fetch a full-resolution frame, find the vehicle in it, read its plate."""
        assert self._snapshots is not None
        stats = self._stats_for(camera)
        try:
            got = await self._snapshots.fetch(cam_row)
            if got is None:
                return
            hires, taken = got
            if hires.shape[1] < detect_shape[1] * platesnap.MIN_GAIN:
                self._snapshots.note_no_gain(camera, hires.shape, detect_shape)
                return
            self._snapshots.note_gain(camera)
            stats.hires_frames += 1
            found = await asyncio.to_thread(platesnap.locate, template, box, detect_shape, hires)
            if self._reader.has_detector:
                # The detector finds the PLATE, so the car does not have to be
                # found again first: search around where it is (or, if the
                # matcher lost it, around where it was, more widely).
                await self._hires_by_detector(
                    st, camera, tracker_id, hires, taken, box, detect_shape, found
                )
                return
            if found is None:
                stats.hires_lost += 1
                return
            hbox, _score = found
            cropped = crop_with_origin(hires, hbox, pad=VEHICLE_CROP_PAD)
            if cropped is None:
                stats.hires_lost += 1
                return
            vehicle, ox, oy = cropped
            hh, hw = hires.shape[:2]
            await self._offer_regions(
                st, camera, tracker_id, vehicle, ox, oy, hw, hh, taken, hires=True
            )
            await self._read_pending(st, tracker_id)
            # Same bar as the detect-frame path: two reads before a live vote,
            # so one misread cannot name a vehicle in a notification.
            if not st.concluded and len(st.reads) >= 2:
                self._tally(st)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("full-resolution plate look failed on %s", camera)

    async def _hires_by_detector(
        self, st: _TrackState, camera: str, tracker_id: int, hires: np.ndarray,
        taken: float, box: tuple[int, int, int, int], detect_shape: tuple[int, int],
        found: Optional[tuple[tuple[float, float, float, float], float]],
    ) -> None:
        stats = self._stats_for(camera)
        hh, hw = hires.shape[:2]
        if found is not None:
            vbox, margins, near = found[0], HIRES_MARGINS_FOUND, 0.15
        else:
            sx, sy = hw / float(detect_shape[1]), hh / float(detect_shape[0])
            vbox = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
            margins, near = HIRES_MARGINS_LOST, 0.6
        # Only plates on or right by THIS vehicle: a wide search area can take
        # in a neighbour's car, and a neighbour's plate is a wrong plate.
        nx1, ny1, nx2, ny2 = _expand(vbox, near, hw, hh)
        cx0, cy0 = (vbox[0] + vbox[2]) / 2.0, (vbox[1] + vbox[3]) / 2.0
        mine: list = []
        # Tight first, then wider: the detector is surest when the vehicle
        # fills the crop, and only needs the wider view when the car has moved
        # since the detect frame (or the matcher lost it).
        for margin in margins:
            rx1, ry1, rx2, ry2 = _expand(vbox, margin, hw, hh)
            region = hires[ry1:ry2, rx1:rx2]
            plates = await asyncio.to_thread(self._reader.detect_blocking, region) or []
            for (x1, y1, x2, y2, score) in plates:
                cx, cy = rx1 + (x1 + x2) / 2.0, ry1 + (y1 + y2) / 2.0
                if nx1 <= cx <= nx2 and ny1 <= cy <= ny2:
                    mine.append(((cx - cx0) ** 2 + (cy - cy0) ** 2, -score, (x1, y1, x2, y2)))
            if mine:
                break
        if not mine:
            stats.hires_lost += 1
            return
        mine.sort()
        await self._offer_regions(
            st, camera, tracker_id, region, rx1, ry1, hw, hh, taken, hires=True,
            regions=[m[2] for m in mine[:2]],
        )
        await self._read_pending(st, tracker_id)
        if not st.concluded and len(st.reads) >= 2:
            self._tally(st)

    async def _read_pending(self, st: _TrackState, tracker_id: int) -> None:
        """OCR every retained shot that has not been read yet, exactly once —
        with every loaded reader, each read a separate vote."""
        for shot in st.buffer.shots(tracker_id, "plate"):
            key = (shot.frame_time, shot.box)
            if key in st.read_shots:
                continue
            st.read_shots.add(key)
            results = await asyncio.to_thread(self._reader.read_all_blocking, shot.crop)
            stats = self._stats_for(st.camera)
            region = self._plate_region()
            for raw, confidence, char_conf in results:
                text = regional_plate(raw, region)
                if not text or len(text) < MIN_PLATE_LENGTH or confidence < OCR_MIN_CONFIDENCE:
                    stats.rejected_reads += 1
                    continue
                stats.reads += 1
                if key in st.hires_keys:
                    stats.hires_reads += 1
                st.reads.append(
                    PlateRead(text=text, confidence=confidence, quality=shot.quality.total,
                              char_conf=char_conf if len(char_conf) == len(text) else ())
                )

    def _tally(self, st: _TrackState) -> None:
        st.vote = vote_plate(st.reads)
        if st.vote is None or self._gallery is None:
            return
        st.match = self._gallery.match_plate(st.vote.text)
        if st.match.matched:
            log.info(
                "recognized %s (%s) on %s from %d read(s)",
                st.match.name, st.vote.text, st.camera, st.vote.reads,
            )
        self._announce(st)

    def _announce(self, st: _TrackState) -> None:
        """Tell the events pipeline what the plate reads, while the event lives.

        Announced even when it matched NOBODY: an unmatched plate still belongs
        in the alert ("plate 7ABC123"), which is the whole point of reading one
        on a vehicle that is not enrolled.
        """
        if self.on_recognition is None or not st.event_fid or st.vote is None:
            return
        if st.vote.confidence < MIN_VOTE_CONFIDENCE:
            return
        try:
            self.on_recognition(
                st.event_fid, "plate",
                name=st.match.name if (st.match and st.match.matched) else "",
                profile_id=st.match.profile_id if (st.match and st.match.matched) else None,
                plate=st.vote.text,
                score=st.match.score if st.match else 0.0,
                alert_mode=st.match.alert_mode if st.match else "default",
            )
        except Exception:
            log.exception("could not announce a plate recognition")

    # ---------- track end ----------

    async def finish(self, camera: str, tracker_id: int) -> None:
        """A vehicle left: vote over every read of it and store the answer.

        If a full-resolution look is still in flight — and it is often the look
        that will actually read the plate — the vote waits for it IN THE
        BACKGROUND. The engine awaits this method inside its frame loop, so
        waiting here would stall detection on every camera for an HTTP round
        trip. The track has already been removed, so a reused tracker id starts
        clean regardless.
        """
        st = self._tracks.pop((camera, tracker_id), None)
        if st is None:
            return
        task = st.hires_task
        if self._wants_replay(st):
            deferred = asyncio.create_task(
                self._conclude_after_replay(st, camera, tracker_id, task),
                name=f"plate-replay-{camera}-{tracker_id}",
            )
            self._finishing.add(deferred)
            deferred.add_done_callback(self._finishing.discard)
            return
        if task is not None and not task.done():
            deferred = asyncio.create_task(
                self._conclude_after(st, camera, tracker_id, task),
                name=f"plate-finish-{camera}-{tracker_id}",
            )
            self._finishing.add(deferred)
            deferred.add_done_callback(self._finishing.discard)
            return
        await self._conclude(st, camera, tracker_id)

    async def _conclude_after(
        self, st: _TrackState, camera: str, tracker_id: int, task: asyncio.Task
    ) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(task), HIRES_FINISH_WAIT_S)
        except asyncio.TimeoutError:
            task.cancel()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the look logs its own failures
            pass
        await self._conclude(st, camera, tracker_id)

    # ---------- reading the recording ----------

    def _replay_on(self) -> bool:
        if self._replay is None or not self._replay.available:
            return False
        if self._settings is None:
            return True
        try:
            cfg = (self._settings.current or {}).get("recognition") or {}
        except Exception:  # noqa: BLE001
            return True
        return bool(cfg.get("plate_replay", True))

    def _wants_replay(self, st: _TrackState) -> bool:
        """Replay unless the live looks already settled the plate."""
        if not self._replay_on() or len(st.path) < 2:
            return False
        v = vote_plate(st.reads) if st.reads else None
        return not (v is not None and v.reads >= SETTLED_READS
                    and v.confidence >= SETTLED_CONFIDENCE)

    async def _conclude_after_replay(
        self, st: _TrackState, camera: str, tracker_id: int,
        task: Optional[asyncio.Task],
    ) -> None:
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), HIRES_FINISH_WAIT_S)
            except asyncio.TimeoutError:
                task.cancel()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                pass
        try:
            await self._replay_reads(st, camera, tracker_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("plate replay failed on %s/%s", camera, tracker_id)
        await self._conclude(st, camera, tracker_id)

    async def _replay_reads(self, st: _TrackState, camera: str, tracker_id: int) -> None:
        """Decode the seconds this vehicle was in view and read every frame."""
        assert self._replay is not None
        stats = self._stats_for(camera)
        path = list(st.path)
        start, end = path[0][0], path[-1][0]
        region = (
            min(b[0] for _, b in path), min(b[1] for _, b in path),
            max(b[2] for _, b in path), max(b[3] for _, b in path),
        )
        frames = await self._replay.frames(camera, start, end, region)
        if not frames:
            return
        stats.replays += 1
        stats.replay_frames += len(frames)
        times = [t for t, _ in path]
        region_name = self._plate_region()
        for fr in frames:
            # Where THIS vehicle was at that moment (nearest sighting), in the
            # crop's pixels — so a parked neighbour's plate in the same crop is
            # not read as this car's.
            i = min(range(len(times)), key=lambda k: abs(times[k] - fr.time))
            bx1, by1, bx2, by2 = path[i][1]
            vb = (bx1 * fr.width - fr.ox, by1 * fr.height - fr.oy,
                  bx2 * fr.width - fr.ox, by2 * fr.height - fr.oy)
            ch, cw = fr.crop.shape[:2]
            nx1, ny1, nx2, ny2 = _expand(vb, 0.3, cw, ch)
            plates = await asyncio.to_thread(self._reader.detect_blocking, fr.crop) or []
            mine = [p for p in plates
                    if nx1 <= (p[0] + p[2]) / 2.0 <= nx2 and ny1 <= (p[1] + p[3]) / 2.0 <= ny2]
            if not mine:
                continue
            x1, y1, x2, y2, _score = max(mine, key=lambda p: p[4])
            strip = fr.crop[y1:y2, x1:x2]
            quality = score_plate(strip)
            if quality.total < MIN_REPLAY_QUALITY:
                continue
            # Offer it too, so the stored candidate crop is the best look of
            # the whole visit rather than whatever the live path managed.
            st.buffer.offer(
                tracker_id=tracker_id, kind="plate", crop_bgr=strip,
                box=(x1, y1, x2, y2), frame_time=fr.time, quality=quality,
                frame_box=(max(0.0, (fr.ox + x1) / fr.width), max(0.0, (fr.oy + y1) / fr.height),
                           min(1.0, (fr.ox + x2) / fr.width), min(1.0, (fr.oy + y2) / fr.height)),
            )
            st.read_shots.add((fr.time, (x1, y1, x2, y2)))
            for raw, confidence, char_conf in await asyncio.to_thread(
                self._reader.read_all_blocking, strip
            ):
                text = regional_plate(raw, region_name)
                if not text or len(text) < MIN_PLATE_LENGTH or confidence < OCR_MIN_CONFIDENCE:
                    stats.rejected_reads += 1
                    continue
                stats.reads += 1
                stats.replay_reads += 1
                st.reads.append(PlateRead(
                    text=text, confidence=confidence, quality=quality.total,
                    char_conf=char_conf if len(char_conf) == len(text) else (),
                ))

    async def wait_idle(self) -> None:
        """Wait for every deferred vote. For tests and orderly shutdown."""
        while self._finishing:
            await asyncio.gather(*list(self._finishing), return_exceptions=True)

    async def _conclude(self, st: _TrackState, camera: str, tracker_id: int) -> None:
        st.concluded = True
        stats = self._stats_for(camera)
        try:
            # The final best shot often arrives on the last pass before the
            # track retires, so sweep once more before voting.
            await self._read_pending(st, tracker_id)
            if not st.reads:
                return
            self._tally(st)
            if st.vote is None or st.vote.confidence < MIN_VOTE_CONFIDENCE:
                stats.votes_discarded += 1
                # A vote nobody would stand behind is not recorded at all: a
                # wrong plate on an event is worse than no plate.
                log.debug(
                    "plate vote on %s/%s discarded (confidence %.2f)",
                    camera, tracker_id,
                    st.vote.confidence if st.vote else 0.0,
                )
                return
            await self._store(st, tracker_id)
            stats.votes_stored += 1
            stats.last_plate = st.vote.text
            stats.last_plate_at = time.time()
        except Exception:
            log.exception("could not finish plate track %s/%s", camera, tracker_id)

    async def _store(self, st: _TrackState, tracker_id: int) -> None:
        assert st.vote is not None
        shot = st.buffer.best(tracker_id, "plate")
        matched = st.match is not None and st.match.matched
        now = time.time()

        # Only against a real event. A plate zone reads cars that never open
        # one (parked in the box, or through it faster than an event
        # confirms); their plate still becomes a candidate below, but a
        # recognition row with no event would belong to nothing.
        if st.event_fid:
            await self._db.conn.execute(
                "INSERT INTO event_recognitions (event_fid, kind, profile_id, name, plate, "
                "score, quality, image_path, created_at) VALUES (?, 'plate', ?, ?, ?, ?, ?, '', ?)",
                (
                    st.event_fid,
                    st.match.profile_id if matched else None,
                    st.match.name if matched else "",
                    st.vote.text,
                    float(st.match.score) if st.match is not None else 0.0,
                    float(shot.quality.total) if shot is not None else 0.0,
                    now,
                ),
            )
        if not matched:
            await self._store_candidate(st, shot, now)
        await self._db.conn.commit()

    async def _store_candidate(
        self, st: _TrackState, shot: Optional[Shot], now: float
    ) -> None:
        """Keep an unmatched plate so the vehicle can be enrolled later.

        Deduped by the PLATE STRING rather than by similarity: two reads of the
        same plate are the same vehicle by definition, and that is a far
        stronger signal than the cosine the face pass has to settle for.
        """
        assert st.vote is not None
        try:
            existing = await (
                await self._db.conn.execute(
                    "SELECT 1 FROM recognition_candidates WHERE kind = 'plate' AND plate = ?",
                    (st.vote.text,),
                )
            ).fetchone()
            if existing is not None:
                return
        except Exception:
            log.exception("plate candidate dedupe failed")

        cur = await self._db.conn.execute(
            "INSERT INTO recognition_candidates (kind, camera, event_fid, embedding, dim, "
            "plate, model_key, image_path, quality, frame_box, best_score, best_profile_id, "
            "created_at) VALUES ('plate', ?, ?, NULL, 0, ?, '', '', ?, ?, ?, ?, ?)",
            (
                st.camera, st.event_fid, st.vote.text,
                float(shot.quality.total) if shot is not None else 0.0,
                encode_frame_box(shot.frame_box if shot is not None else None),
                float(st.match.score) if st.match is not None else 0.0,
                None, now,
            ),
        )
        candidate_id = cur.lastrowid
        if shot is not None:
            name = await asyncio.to_thread(self._write_crop, candidate_id, shot.crop)
            if name:
                await self._db.conn.execute(
                    "UPDATE recognition_candidates SET image_path = ? WHERE id = ?",
                    (name, candidate_id),
                )
        await self._prune()

    def _write_crop(self, candidate_id: int, crop: np.ndarray) -> str:
        import cv2

        try:
            self._images_dir.mkdir(parents=True, exist_ok=True)
            name = f"plate-{candidate_id}.jpg"
            ok = cv2.imwrite(str(self._images_dir / name), crop,
                             [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            return name if ok else ""
        except Exception:
            log.exception("could not write plate crop %d", candidate_id)
            return ""

    async def _prune(self) -> None:
        try:
            row = await (
                await self._db.conn.execute(
                    "SELECT COUNT(*) AS n FROM recognition_candidates WHERE kind = 'plate'"
                )
            ).fetchone()
            excess = (row["n"] or 0) - MAX_CANDIDATES
            if excess <= 0:
                return
            victims = await (
                await self._db.conn.execute(
                    "SELECT id, image_path FROM recognition_candidates WHERE kind = 'plate' "
                    "ORDER BY quality ASC, created_at ASC LIMIT ?",
                    (excess,),
                )
            ).fetchall()
            ids = [r["id"] for r in victims]
            await self._db.conn.execute(
                f"DELETE FROM recognition_candidates WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            for r in victims:
                if r["image_path"]:
                    (self._images_dir / r["image_path"]).unlink(missing_ok=True)
        except Exception:
            log.exception("plate candidate prune failed")

    # ---------- background ----------

    async def run(self, settings: Any) -> None:
        """Load/release the OCR as settings.recognition.enabled changes."""
        while True:
            try:
                self._settings = settings
                cfg = (settings.get() or {}).get("recognition") or {}
                enabled = bool(cfg.get("enabled"))
                self._shots, self._shot_gap = shot_params(cfg)
                if enabled and not self._reader.ready:
                    if await self._reader.load():
                        await self.reload_gallery()
                elif not enabled and self._reader.ready:
                    log.info("recognition disabled — releasing the plate OCR")
                    self._reader.close()
                    self._cancel_hires()
                    self._tracks.clear()
                    if self._snapshots is not None:
                        await self._snapshots.close()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("plate maintenance tick failed")
            await asyncio.sleep(RUN_INTERVAL_S)

    # ---------- lifetime ----------

    def _cancel_hires(self, camera: Optional[str] = None) -> None:
        for (cam, _tid), st in self._tracks.items():
            if camera is not None and cam != camera:
                continue
            if st.hires_task is not None and not st.hires_task.done():
                st.hires_task.cancel()

    def forget_camera(self, camera: str) -> None:
        self._cancel_hires(camera)
        for key in [k for k in self._tracks if k[0] == camera]:
            del self._tracks[key]
        self._stats.pop(camera, None)
        if self._snapshots is not None:
            self._snapshots.forget(camera)

    def status(self) -> dict[str, Any]:
        return {
            "live_tracks": len(self._tracks),
            "ready": self._reader.ready,
            "vehicles": len(self._gallery) if self._gallery is not None else 0,
            # Whether full-resolution looks are on, and per camera how each
            # stage is doing — see _CameraStats for how to read it.
            "hires": self._hires_on(),
            # How plates are being FOUND (detector or classic) and how many
            # readers read them, so "is the new detector actually in use?" has
            # an answer without reading logs.
            "reader": self._reader.status(),
            "cameras": {cam: s.report() for cam, s in sorted(self._stats.items())},
            "snapshots": self._snapshots.status() if self._snapshots is not None else {},
            # Recording replays: whether they decode on the GPU first, and how
            # the decodes have actually gone.
            "replay": (self._replay.status()
                       if self._replay is not None and hasattr(self._replay, "status") else {}),
        }
